def generate_tryon(data_path):
  # im size
  fine_width = 192
  fine_height = 256
  # keypoint size
  radius = 5
  grid_size = 3

  # transform input images
  transform_im = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, ), (0.5,))
  ])

  ### generate cloth, cloth mask, and agnostic image for gmm
  ## agnostic requires im_h, shape, pose_map
  # im_h requires image and image_parse
  im_parse = Image.open(os.path.join(data_path, 'image_parse.png'))
  parse_array = np.array(im_parse)
  parse_shape = (parse_array > 0).astype(np.float32)
  parse_head = (parse_array == 1).astype(np.float32) + \
          (parse_array == 2).astype(np.float32) + \
          (parse_array == 4).astype(np.float32) + \
          (parse_array == 13).astype(np.float32)
  phead = torch.from_numpy(parse_head)
  im = Image.open(os.path.join(data_path, 'image.jpg'))

  im = transform_im(im)
  im_h = im * phead - (1 - phead)   # (3, 256, 192)
  im = im.cuda()
  im_h = im_h.cuda()


  # shape requires im_parse
  parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
  parse_shape = parse_shape.resize((fine_width//16, fine_height//16), Image.BILINEAR)
  parse_shape = parse_shape.resize((fine_width, fine_height), Image.BILINEAR)
  
  shape = transform_im(parse_shape)   # (1, 256, 192)
  shape = shape.cuda()


  # pose_map requires pose keypoints
  with open(os.path.join(data_path, 'pose.json'), 'r') as f:
      pose_label = json.load(f)
      pose_data = pose_label['people'][0]['pose_keypoints']
      pose_data = np.array(pose_data)
      pose_data = pose_data.reshape((-1,3))

  point_num = pose_data.shape[0]
  pose_map = torch.zeros(point_num, fine_height, fine_width)
  r = radius
  im_pose = Image.new('L', (fine_width, fine_height))
  pose_draw = ImageDraw.Draw(im_pose)
  for i in range(point_num):
      one_map = Image.new('L', (fine_width, fine_height))
      draw = ImageDraw.Draw(one_map)
      pointx = pose_data[i,0]
      pointy = pose_data[i,1]
      if pointx > 1 and pointy > 1:
          draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
          pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
      one_map = transform_im(one_map)
      pose_map[i] = one_map[0]
  # pose_map: (18, 256, 192)

  agnostic = torch.cat([shape, im_h, pose_map])   # (22, 256, 192)
  agnostic = torch.unsqueeze(agnostic, 0)   # add batch size dim
  agnostic = agnostic.cuda()
  
  
  ## process cloth and cloth_mask
  c = Image.open(os.path.join(data_path, 'cloth.jpg'))
  cm = Image.open(os.path.join(data_path, 'cloth_mask.jpg'))

  c = transform_im(c)   # (3, 256, 192)
  cm_array = np.array(cm)
  cm_array = (cm_array >= 128).astype(np.float32)
  cm = torch.from_numpy(cm_array)
  cm.unsqueeze_(0)   # (1, 256, 192)

  c = torch.unsqueeze(c, 0)
  cm =  torch.unsqueeze(cm, 0)
  c = c.cuda()
  cm = cm.cuda()

  ### grid image for generating warping
  im_g = Image.open(os.path.join(data_path,'grid.png'))
  im_g = transform_im(im_g)   # (3, 256, 192)
  im_g = torch.unsqueeze(im_g, 0)
  im_g = im_g.cuda()


  ### init gmm model, generate warping
  gmm = GMM()
  # gmm.load_state_dict(torch.load(gmm_ckpt_path))
  gmm.cuda()



  with torch.no_grad():
    grid, theta = gmm(agnostic, c)   # grid (256, 192, 2), theta (18)
    warped_cloth = F.grid_sample(c, grid, padding_mode='border')   # (2, 356, 192)
    warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')   # (1, 256, 192)
    warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')   # (3, 256, 192)
  
  ### tom
  tom = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
  tom.load_state_dict(torch.load(tom_ckpt_path))
  tom.cuda()

  with torch.no_grad():
    outputs = tom(torch.cat([agnostic, c],1))
    p_rendered, m_composite = torch.split(outputs, 3,1)
    p_rendered = F.tanh(p_rendered)
    m_composite = F.sigmoid(m_composite)
    p_tryon = c * m_composite+ p_rendered * (1 - m_composite)   # (3, 256, 192)

  
  return p_tryon
